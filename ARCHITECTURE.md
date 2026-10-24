# Decode & Detect — XRPL AI Hack: Architecture & Team Plan

**Event:** Sat 24 Oct 2026, IDEALondon · submissions close **16:30** · check-in by 10:30
**Task:** one suspicion score per test transaction (synthetic XRPL-style money laundering), minimise false positives.
**Metric:** TBC at briefing (plan for ROC-AUC / PR-AUC — both are rank-based, so the ensemble below works for either).

---

## 1. System architecture

```
data/raw/*                      (organiser files — NEVER committed)
        │
        ▼
src/data/load.py                schema.yaml maps organiser column names → canonical names
        │   tx (one row per transaction), accounts (optional), train labels (if given)
        ▼
┌───────────────────────────────────────────────────────────────┐
│ FEATURE LAYER  — each module: build(tx, accounts) -> DataFrame │
│   indexed by tx_id, columns prefixed by module name            │
│                                                                 │
│  src/features/tx.py      tx_*     per-transaction features      │
│  src/features/account.py acct_*   sender/receiver aggregates    │
│  src/features/graph.py   graph_*  networkx-derived features     │
│  src/features/rules.py   rule_*   typology flags (0/1 + score)  │
└───────────────────────────────────────────────────────────────┘
        │   pd.concat(axis=1) → data/processed/features.parquet
        ▼
src/model/train.py             LightGBM (if labels) + IsolationForest (always)
src/model/ensemble.py          rank-average of model scores + rule score
        │
        ▼
src/model/predict.py  →  outputs/predictions.csv  (tx_id, suspicion_score ∈ [0,1])
        │
        ▼
scripts/submit_log.py          append leaderboard result to outputs/leaderboard_log.csv
```

**Design rules**
- Everything keyed on `tx_id`. Feature modules never join on anything else.
- Each feature module is independent — parallel work, no merge conflicts.
- `run.sh` runs the whole thing from raw data to `predictions.csv` in one command, deterministically (`SEED=42` everywhere, `n_jobs=1` for final run).
- Two modes, selected automatically by `load.py`: **supervised** (train labels present) and **unsupervised** (no labels → IsolationForest + rules only).

---

## 2. Data contract (real organiser format, confirmed 24 Oct)

Organiser data = one folder per split (`train/`, `validation/`, `test/`), each with
`transactions.parquet`, `transfers.parquet` (1–3 legs per tx), `blocks.parquet` (unused),
`market_rates.csv` (hourly USD prices); `train/` also has `labels.csv`.
Locally they are symlinked into `data/raw/{train,validation,test}` (never committed).
**Wallets are disjoint between splits → features are always computed within one split.**

`src/data/load.py::load_split(dir)` returns `(tx, legs)`.

**`tx`: one row per transaction**, sorted by `(timestamp, tx_id)`:

| Column | Type | Meaning / source |
|---|---|---|
| `tx_id` | str | `TRANSACTION_HASH` |
| `timestamp` | datetime UTC | `TIMESTAMP` (unix s) |
| `ts` | int | unix seconds (for windows) |
| `sender` | str | `ACCOUNT` (= first leg's `FROM_ADDRESS`) |
| `receiver` | str | `Destination` in `EXTRA_TRANSACTION_FIELDS` (= last leg's `TO_ADDRESS`) |
| `asset_out`, `amount_out` | str, float | delivered asset (`XRP` or `issuer:CUR`) and units received |
| `asset_in`, `amount_in` | str, float | first executed leg: what the sender actually paid |
| `value_usd` | float | delivered value at the hourly reference price — **use this to compare amounts** |
| `value_in_usd` | float | sender's actual cost in USD; `value_in_usd - value_usd` = conversion spread |
| `n_legs` | int | 1 = direct, 2 = one provider, 3 = XRP bridge |
| `is_conversion` | 0/1 | `asset_in != asset_out` |
| `providers` | str | `;`-joined liquidity providers used (empty if direct) |
| `fee` | float | XRP fee |
| `label` | 0/1 | `is_suspicious`, train only — **never reaches features** |
| `meta_family`, `meta_difficulty`, `meta_scenario_id`, `meta_cohort_id` | str | train only, for CV grouping / reporting — **never features** |

**`legs`: one row per transfer leg**: `tx_id, leg, from_addr, to_addr, asset, amount, value_usd, transfer_type`.

Feature modules receive `tx` without `label`/`meta_*`, plus `legs`, for ONE split.
Forbidden as features: wallet addresses, hashes, ledger/sequence numbers, calendar date
(test is a different week), anything `meta_*`.

---

## 3. Typologies to encode (what the synthetic generator probably planted)

| Typology | Signal | Feature home |
|---|---|---|
| Structuring / smurfing | many sub-threshold amounts, same sender, short window | account + rules |
| Pass-through / mule | in-volume ≈ out-volume, short holding time | account |
| Layering chain | A→B→C→D same-ish amount within minutes | graph |
| Round-tripping / cycle | money returns to origin (cycle length 2–5) | graph |
| Fan-in → fan-out | high in-degree then high out-degree | graph |
| Currency hop | XRP→IOU→XRP to obscure trail | tx + rules |
| Round amounts | amount % 1000 == 0 etc. | tx |
| Velocity burst | dormant account suddenly very active | account |
| New-account risk | tx soon after first appearance | account |

---

## 4. Team split (5 people)

| # | Role | Owner | Owns | Delivers by |
|---|---|---|---|---|
| **P1** | **Captain / Pipeline & Submissions** | ___ | `load.py`, `schema.yaml`, `run.sh`, `predict.py`, submission CSV format, **all leaderboard submissions**, `leaderboard_log.csv` | baseline submitted by 11:45 |
| **P2** | **Transaction + Account features** | ___ | `features/tx.py`, `features/account.py` | v1 by 12:30, v2 by 14:00 |
| **P3** | **Graph features** | ___ | `features/graph.py` (networkx) | v1 by 13:00, v2 by 14:30 |
| **P4** | **Modelling & evaluation** | ___ | `model/train.py`, `model/ensemble.py`, CV, feature importance, leaderboard-vs-CV sanity check | model v1 by 13:00, final by 15:00 |
| **P5** | **Rules, EDA, Repro & Report** | ___ | early EDA + typology hunting → `features/rules.py`; then from 14:30: README disclosures, LICENSE, clean-clone reproduction test, report/deck, commit tag | report draft 15:45, final 16:15 |

**Interface contract (do not break):**
```python
def build(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Returns DataFrame indexed by tx_id. Columns prefixed e.g. 'graph_'. No NaN-only columns."""
```

**Branching:** `develop` is the default branch; each person on `feat/<name>`; P1 merges. Small commits, merge every ~45 min. P4 owns `features.parquet` regeneration after each merge.

---

## 5. Timeline

| Time | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| 10:30 | Briefing — everyone writes down metric, format, submission cap, schema |||||
| 11:00 | load.py + schema | tx features v1 | graph build + degree feats | CV harness, IsolationForest baseline | EDA: distributions, look for planted patterns |
| 11:45 | **Submit baseline** | | | | |
| 12:30 | merge, regen features | account aggregates | cycles / chains | LightGBM v1 | rules.py v1 |
| 13:00 | **Submit v1** | | | | |
| 13:30 | merge | velocity / burst | same-amount-forwarding, neighbour score propagation | ensemble, importance → feedback to P2/P3 | rules v2, README skeleton |
| 14:30 | **Submit v2** | polish | polish | tune, final ensemble | clean-clone repro test |
| 15:00 | **Submit v3** | help P5 | help P5 | **freeze model** | report / deck |
| 15:45 | final checks | | | | report done |
| 16:15 | **FINAL SUBMIT, tag commit `v-final`** | | | | |

Rule: a submission every ~75 min, never later than 16:15. Model frozen at 15:00; everything after is reproducibility and report.

---

## 6. Submission checklist (from T&Cs §3.3, §3.6, §5.3, §4.4)

- [ ] `outputs/predictions.csv` — one score per test `tx_id`, correct column names
- [ ] Public repo, `LICENSE` = MIT, `README.md` complete
- [ ] Commit hash / tag identified in submission form
- [ ] `run.sh` reproduces predictions from a clean clone (P5 verifies on a second laptop)
- [ ] `requirements.txt` pinned; compute requirements stated (laptop CPU, ~N min)
- [ ] README lists every third-party library + licence (§6.1)
- [ ] README states what was pre-existing vs built today (§3.4)
- [ ] README discloses AI coding assistants used (model/provider) — no LLM in the scoring path (§3.3)
- [ ] No dataset in git (unless organisers say it's allowed), no credentials (§4.3)
- [ ] Report/deck covers: method, results, **uncertainty, false positives, data limitations, bias; not a compliance tool** (§4.4)
- [ ] Team name, captain, all 5 members listed; everyone accepted T&Cs individually

---

## 7. Questions for the 10:30 briefing

1. Exact metric? Tie-break?
2. Are train labels provided? Proportion positive?
3. Submission cap or rate limit on the leaderboard?
4. Required CSV columns / ID field name?
5. Can the dataset be committed to the public repo?
6. Are account-creation timestamps / account metadata provided?
7. Any time window to expect (hours? months?) — affects velocity feature windows.
