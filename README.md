# Turac — Decode & Detect: XRPL suspicion scoring

## 1. What this is

A hackathon entry that assigns a **suspicion score in [0, 1]** to each transaction in a
**synthetic**, XRPL-style dataset supplied by the Decode & Detect organisers.

> **Disclaimer.** This is a one-day research prototype built on synthetic data. It is
> **not a real fraud detector and not a compliance tool**, and must not be used to make
> decisions about real people, accounts or transactions.

## 2. Team

| Role | Name |
|---|---|
| Team name | [team name] |
| Captain (P1) | [name] |
| P2 — tx + account features | [name] |
| P3 — graph features | [name] |
| P4 — modelling | [name] |
| P5 — rules, repro, report | [name] |

## 3. How to reproduce

```bash
git clone [repo url] && cd [repo]
# Organiser data is not in git. Put (or symlink) the split folders here:
#   data/raw/train/        <- training package train/   (incl. labels.csv)
#   data/raw/validation/   <- training package validation/
#   data/raw/test/         <- the test download (transactions.parquet, transfers.parquet, ...)
./run.sh                 # -> outputs/predictions.csv (TRANSACTION_HASH,score), validated
./run.sh --sample        # same pipeline on the bundled synthetic mini-splits
make cv                  # grouped-CV report on train only (no test data needed)
make zip                 # outputs/code.zip for upload (tracked files only, no data)
```

- Runtime: ~2.5 min for `./run.sh` (5-seed bagging) on a laptop CPU ([CPU model, RAM]). No GPU, no network access.
- Python 3.13; exact versions pinned in `requirements.txt`.
- Deterministic: `SEED = 42` (`src/config.py`), `n_jobs = 1`. `make test` checks that two
  runs give byte-identical predictions.
- macOS: LightGBM needs OpenMP (`brew install libomp`). Without it the pipeline falls back
  to scikit-learn's `HistGradientBoostingClassifier` and logs a warning; the final
  submission was produced with LightGBM 4.7.0.

## 4. Method overview

Each split is loaded into one row per payment, valued in USD at the organisers' hourly
reference prices (`src/data/load.py`). Feature modules (`src/features/`: starter-notebook
rules, transaction, account, graph, rule flags) are computed **within each split**
(wallets are disjoint between splits) → gradient-boosted trees → `score`.
Validation: repeated 5-fold CV **grouped by scenario**, reported per difficulty.
See [report/REPORT.md](report/REPORT.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## 5. Submitted commit

`[hash]` (tag `v-final`)

## 6. Pre-existing vs built during the event (§3.4)

- **Pre-existing:** the planning docs `ARCHITECTURE.md` and `CLAUDE_CODE_BRIEF.md`, written
  before the event. `src/features/starter.py` is a port of the organisers'
  `starter_notebook.ipynb` features.
- **Built during the event (24 Oct 2026):** everything in `src/`, `scripts/`, `tests/`,
  `config/` and `report/`.

## 7. Third-party components & licences (§6.1)

| Component | Version | Licence | Use |
|---|---|---|---|
| pandas | 3.1.0 | BSD-3-Clause | data handling |
| numpy | 2.5.4 | BSD-3-Clause | numerics |
| scikit-learn | 1.9.1 | BSD-3-Clause | IsolationForest, CV, metrics |
| scipy | 1.18.1 | BSD-3-Clause | ranking, pagerank backend |
| lightgbm | 4.7.0 | MIT | gradient boosting |
| networkx | 3.7 | BSD-3-Clause | graph features |
| PyYAML | 6.0.3 | MIT | config |
| joblib | 1.6.0 | BSD-3-Clause | model persistence |
| pyarrow | 26.0.0 | Apache-2.0 | parquet |
| pytest | 9.1.1 | MIT | tests |
| matplotlib | 3.11.2 | Matplotlib (PSF-based) | EDA plots |
| jupyter | 1.1.1 | BSD-3-Clause | EDA notebook |
| [other] | | | |

## 8. AI assistance disclosure (§3.3)

**Tool and models.** Claude Code (Anthropic's CLI agent) with model `claude-fable-5-1`
(earlier in the day `claude-opus-5-5`), provider Anthropic. It was used for code generation,
EDA scripts and experiment analysis, driven interactively by the team. Settings: default
Claude Code CLI, no custom system prompt. The only project-level instructions were
`CLAUDE.md` and `ARCHITECTURE.md` in this repo. [other tools, if any]

**Kinds of prompts given** (paraphrased; the interactive session is not archived verbatim):

- Scaffold the repo per `ARCHITECTURE.md` (feature-module contract, config, run script).
- Adapt the data loader to the organisers' parquet format (transactions, transfers, market rates).
- Port the starter notebook features and build the `tx`, `account`, `flow`, `motif`, `shapes`,
  `graph` and `rules` feature modules.
- Set up scenario-grouped repeated CV with paired fold comparison and a train/test drift (KS) check.
- Analyse errors: which legitimate patterns crowd the top 1-3%, and why medium/hard cases are missed.
- Investigate why a CV gain (learned scenario links) did not transfer to the leaderboard.
- Try a medium-case specialist model and blends (rejected), and burst-level score post-processing.
- Write tests (including deterministic reruns) and draft `report/REPORT.md` and this README.

**No LLM is used at inference time and no LLM output is in the scoring path.** Prompts and
code were reviewed and run by the team, and every number in the report comes from the
repo's experiment logs.

## 9. External services / data used

None. [or list]

## 10. Limitations, false positives, bias (§4.4)

- Synthetic data: patterns the generator planted may not resemble real laundering. A high
  score here does not show the method works on real financial crime.
- False positives: legitimate relay hubs, busy background wallets and decoy motifs of the
  same shape crowd the top 1-3% of scores.
- Not caught: medium cases (AP about 5-8), hard relays (about 1) and the 25 ambiguous
  payments (labels independent of observable features).
- Features are computed within one split (not transductive across train and test) and use
  the whole week, forward windows included: fine offline, not usable in real time.
- Bias: one generator style and issuer pair in train; no asset, wallet or date is a feature.

## 11. Licence

MIT — see [LICENSE](LICENSE).
