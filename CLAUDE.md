# Project notes for Claude Code

Hackathon repo: suspicion scoring of synthetic XRPL-style txs. `ARCHITECTURE.md` is the
source of truth for module boundaries, roles (P1–P5) and timeline.

- Run: `./run.sh` (real data in `data/raw/`) or `./run.sh --sample`. Tests: `make test`.
- Feature contract: `build(tx, accounts, cfg) -> DataFrame` indexed by `tx_id`, columns
  prefixed with the module name, finished with `validate(df, PREFIX)` (`src/features/base.py`).
- Deterministic: use `cfg["seed"]` / `src.config.SEED` for anything random; `n_jobs=1`.
- No LLM calls in the scoring path. No data, models or secrets in git.
- Thresholds and weights go in `config/params.yaml`, not in code.
