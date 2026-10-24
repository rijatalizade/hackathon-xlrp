# Claude Code brief — build the project base

You are scaffolding a Python repo for a one-day hackathon: **suspicion scoring of synthetic XRPL-style transactions** (money-laundering detection). Five people will work on it in parallel from ~11:00; submissions close 16:30. Read `ARCHITECTURE.md` first — it is the source of truth for module boundaries and interfaces.

Build the **skeleton with working stubs**, not the final model. Every module must run end-to-end on a tiny synthetic sample so people can plug in real logic without fighting the pipeline.

## Non-negotiables (from the hackathon T&Cs)

1. **Deterministic.** Global `SEED = 42` in `src/config.py`; pass it to every random component. Final run uses `n_jobs=1`. No LLM calls anywhere in the scoring path.
2. **No data in git.** `.gitignore` must exclude `data/`, `outputs/`, `*.parquet`, `*.csv` (except `outputs/leaderboard_log.csv` and `data/sample/`), `.env`, `*.pkl`.
3. **No secrets.** Never write API keys or credentials into any file.
4. **One command reproduces everything:** `./run.sh` → `outputs/predictions.csv`.
5. **Open source:** `LICENSE` = MIT, year 2026, holder "Decode & Detect hackathon team".
6. **Disclosure-ready README** with the sections listed below, left as fill-in templates.

## Repo layout to create

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── run.sh
├── Makefile
├── config/
│   ├── schema.yaml          # organiser column → canonical name mapping (placeholder values)
│   └── params.yaml          # feature windows, model hyperparams, ensemble weights
├── data/
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── sample/              # tiny generated dataset so the pipeline runs without real data
├── outputs/
│   ├── .gitkeep
│   └── leaderboard_log.csv  # header only: timestamp,commit,submission_name,cv_score,lb_score,notes
├── notebooks/
│   └── eda.ipynb            # empty-ish notebook that loads data via src.data.load
├── report/
│   └── REPORT_TEMPLATE.md
├── scripts/
│   ├── make_sample_data.py  # generates data/sample/ with planted typologies
│   ├── validate_submission.py
│   └── submit_log.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py          # build() signature + helper to validate output
│   │   ├── tx.py
│   │   ├── account.py
│   │   ├── graph.py
│   │   └── rules.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── cv.py
│   │   ├── train.py
│   │   ├── ensemble.py
│   │   └── predict.py
│   └── pipeline.py          # orchestrates: load → features → train → predict
└── tests/
    ├── test_features_contract.py
    ├── test_pipeline_smoke.py
    └── test_determinism.py
```

## Specifications

### `src/config.py`
`SEED = 42`, `ROOT`, path constants, `load_params()` reading `config/params.yaml`, `load_schema()` reading `config/schema.yaml`.

### `src/data/load.py`
- `load_raw(cfg) -> (tx: DataFrame, accounts: DataFrame|None)`: reads every CSV/parquet in `data/raw/` (or `data/sample/` when `--sample`), renames columns via `schema.yaml`, parses `timestamp` to UTC, sorts by timestamp, enforces `tx_id` uniqueness.
- `detect_mode(tx) -> "supervised" | "unsupervised"` based on presence of non-null `label` in train split.
- `split(tx) -> (train, test)` using `split` column, or file names if organisers ship separate files.
- Log the schema it actually found and any unmapped columns as `extra_*`.

### `src/features/base.py`
```python
def build(tx, accounts, cfg) -> pd.DataFrame  # the contract
def validate(df, prefix) -> pd.DataFrame      # index == tx_id, all columns start with prefix, drop all-NaN cols, cast bool→int
```

### `src/features/tx.py` (prefix `tx_`) — implement these for real, they are cheap:
`tx_amount`, `tx_log_amount`, `tx_is_round_100/1000/10000`, `tx_fee`, `tx_hour`, `tx_dayofweek`, `tx_is_conversion` (currency != XRP or tx_type in conversion set), `tx_amount_z_sender` (z-score vs sender's own amounts), `tx_secs_since_sender_prev`, `tx_secs_since_receiver_prev`.

### `src/features/account.py` (prefix `acct_`) — implement:
For both sender and receiver (columns `acct_s_*` and `acct_r_*`): `n_tx_total`, `n_tx_1h`, `n_tx_24h` (rolling, point-in-time, no leakage from the future), `vol_in`, `vol_out`, `inout_ratio`, `n_distinct_counterparties`, `age_secs` (since first seen), `mean_gap_secs`, `burstiness` (std/mean of gaps).

### `src/features/graph.py` (prefix `graph_`) — implement degree features for real, stub the rest with TODOs and clear docstrings:
Build a `networkx.DiGraph` from train+test (no labels used). Per tx, attach sender/receiver: `in_degree`, `out_degree`, `pagerank`, `weakly_cc_size`. Stubs with signatures: `short_cycle_member(k=2..5)`, `chain_depth_same_amount(window_min, tol)`, `neighbour_mean_score` (second-pass feature, takes a prior score vector).

### `src/features/rules.py` (prefix `rule_`) — implement:
`rule_structuring` (sender has ≥N txs below threshold T within window W), `rule_passthrough` (receiver forwards ≥80% of amount within W), `rule_round_amount`, `rule_velocity_burst`, and `rule_score` = weighted sum from `params.yaml`. Thresholds in `params.yaml`, not hard-coded.

### `src/model/`
- `cv.py`: stratified 5-fold, returns OOF scores + metric (`roc_auc` and `average_precision`; metric name from `params.yaml`).
- `train.py`: `train_lgbm(X, y, params, seed)` and `train_iforest(X, seed)`; save with joblib to `outputs/models/`.
- `ensemble.py`: `rank_average({name: scores}, weights)` → scores scaled to [0,1].
- `predict.py`: loads models, builds `outputs/predictions.csv` with columns from `params.yaml` (`id_col`, `score_col`), asserts one row per test `tx_id`, no NaN, scores in [0,1].

### `src/pipeline.py`
CLI: `python -m src.pipeline --mode auto|supervised|unsupervised --sample --skip-graph`. Steps: load → build all features (parallel-safe, each wrapped in try/except that logs and continues so one broken module doesn't block a submission) → save `features.parquet` → train (if supervised) → predict → write CSV → print CV metric.

### `scripts/make_sample_data.py`
Generate ~5,000 transactions, ~400 accounts, 48h window, with planted: a structuring sender, two pass-through mules, one 4-hop layering chain, one 3-cycle. Write `data/sample/transactions.csv` with `label` and `split` columns. This is only so the pipeline runs; mark it clearly as not the organiser data.

### `scripts/validate_submission.py`
Checks `outputs/predictions.csv` against test ids: row count, uniqueness, column names, no NaN, range. Exit non-zero on failure. `run.sh` calls it last.

### `scripts/submit_log.py`
`python scripts/submit_log.py --name v1 --cv 0.91 --lb 0.89 --notes "..."` appends a row with current git commit hash.

### `run.sh`
```bash
set -euo pipefail
python -m venv .venv 2>/dev/null || true; source .venv/bin/activate
pip install -q -r requirements.txt
python -m src.pipeline --mode auto "$@"
python scripts/validate_submission.py
```

### `Makefile`
Targets: `setup`, `sample` (make sample data), `features`, `run`, `test`, `clean`, `submit-check`.

### `requirements.txt` (pin exact versions you install)
pandas, numpy, scikit-learn, lightgbm, networkx, pyyaml, joblib, pyarrow, pytest, matplotlib, jupyter.

### `tests/`
- `test_features_contract.py`: every module in `src.features` returns index==tx_id, correct prefix, no duplicate columns.
- `test_pipeline_smoke.py`: `--sample --mode auto` runs end-to-end in <60s and produces a valid CSV.
- `test_determinism.py`: two sample runs produce byte-identical `predictions.csv`.

### `README.md` template sections (leave bracketed placeholders)
1. What this is (synthetic challenge disclaimer — not a real fraud detector, not a compliance tool)
2. Team: name, captain, members
3. How to reproduce (clone, `./run.sh`, expected runtime, hardware)
4. Method overview (link to report)
5. Submitted commit: `[hash]`
6. **Pre-existing vs built during the event** (§3.4)
7. **Third-party components & licences** (§6.1) — table
8. **AI assistance disclosure** (§3.3): "Claude Code (Anthropic, claude-fable-5-1) and [others] were used for code generation. No LLM is used at inference time."
9. External services / data used: none (or list)
10. Limitations, false positives, bias (§4.4)
11. Licence: MIT

### `report/REPORT_TEMPLATE.md`
Headings: Problem · Data & assumptions · Method (features, model, ensemble) · Validation & results (CV vs leaderboard) · What the model flags and why (top features, examples) · Uncertainty & false positives · Data limitations & bias · Reproducibility · What we'd do next.

## Working style

- Build in this order: config → load → sample data → base/tx features → pipeline + predict → tests → remaining feature modules → README/report templates. Run `make test` before finishing.
- Clear, short docstrings on every public function stating the contract. Add `# TODO(P3)`-style owner tags on stubs matching the roles in `ARCHITECTURE.md`.
- Commit at the end with message `Scaffold pipeline, feature contract, sample data, tests`.
- Do not fetch any external data, do not call any APIs, do not install anything beyond `requirements.txt`.
- When done, print: how to run on real data (drop files in `data/raw/`, edit `config/schema.yaml`, `./run.sh`), and a list of every stub still marked TODO with its owner.
