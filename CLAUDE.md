# Project notes for Claude Code

Hackathon repo (team Turac): suspicion scoring of synthetic XRPL-style payments.
`ARCHITECTURE.md` is the source of truth for the data contract (§2), roles (P1–P5) and
timeline. Organiser rules: README.md in the training package.

- Data: `data/raw/{train,validation,test}/` (organiser split folders, never committed).
- Run: `./run.sh` (test → `outputs/predictions.csv`), `make cv` (CV only), `make test`.
- Feature contract: `build(tx, legs, cfg) -> DataFrame` indexed by `tx_id`, columns
  prefixed with the module name, finished with `validate(df, PREFIX)` (`src/features/base.py`).
  Features are built within ONE split; never use wallet addresses, hashes, issuers or
  calendar dates as feature values; never touch `label` / `meta_*`.
- Judge changes on grouped CV (`meta_scenario_id`), not single splits or the leaderboard;
  a gain smaller than the fold std is noise. `scripts/drift_check.py` for train/test shift.
- Deterministic: use `cfg["seed"]` / `src.config.SEED` for anything random; `n_jobs=1`.
- No LLM calls in the scoring path. No data, models or secrets in git.
- Thresholds and weights go in `config/params.yaml`, not in code.
