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
# put the organiser files in data/raw/ (they are not in git), check config/schema.yaml
./run.sh                 # -> outputs/predictions.csv, validated
./run.sh --sample        # runs on the bundled synthetic sample instead
```

- Expected runtime: [~N min] on [laptop CPU model, RAM]. No GPU needed.
- Python [3.x]; exact versions pinned in `requirements.txt`.
- Deterministic: `SEED = 42` (`src/config.py`), `n_jobs = 1`. `make test` checks that two
  runs give byte-identical predictions.
- macOS: LightGBM needs OpenMP (`brew install libomp`). Without it the pipeline falls back
  to scikit-learn's `HistGradientBoostingClassifier` and logs a warning; the final
  submission was produced with [LightGBM / HistGradientBoosting].

## 4. Method overview

Feature modules (`src/features/`: transaction, account, graph, rule flags) → LightGBM
(if labels are given) + IsolationForest → rank-average ensemble with the rule score.
See [report/REPORT.md](report/REPORT.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## 5. Submitted commit

`[hash]` (tag `v-final`)

## 6. Pre-existing vs built during the event (§3.4)

- **Pre-existing:** [planning docs ARCHITECTURE.md / CLAUDE_CODE_BRIEF.md written before
  the event; anything else].
- **Built during the event:** [everything in src/, scripts/, tests/, config/, report/].

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

Claude Code (Anthropic, claude-fable-5-1) and [others] were used for code generation.
No LLM is used at inference time.

## 9. External services / data used

None. [or list]

## 10. Limitations, false positives, bias (§4.4)

- [Synthetic data: patterns the generator planted may not resemble real laundering.]
- [False-positive profile: which legitimate behaviours score high (e.g. busy exchanges,
  payroll-like fan-out, round-number merchants).]
- [Graph features are transductive (built from train + test); rules use forward windows.]
- [Bias: which accounts/currencies are over- or under-represented; threshold choices.]

## 11. Licence

MIT — see [LICENSE](LICENSE).
