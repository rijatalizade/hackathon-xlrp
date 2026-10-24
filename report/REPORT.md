# Turac — Decode & Detect report

**Read this first.** The data is synthetic. This is a one-day research prototype, not a real fraud detector and not a compliance tool. A high score on this ledger does not show that the method works on real financial crime, and it must not be used to judge real people, accounts or transactions.

## Problem

Give every transaction in the test set (40,101 payments) a suspicion score in [0, 1] so that the planted money-laundering transactions rank first. The score is 100 x average precision (AP). AP punishes false positives high in the ranking, so the goal is a clean top of the list.

## Data & assumptions

- **Train:** 40,131 payments, 700 wallets, 7 days, 1,250 suspicious (3.1%): 800 easy, 225 medium, 225 hard. **Test:** 40,101 payments from a different week, with new wallets and a different token issuer. The organisers say the test set has batch variations that training lacks.
- We keep one row per transaction (sender, receiver, delivered value) and value it in USD at the organisers' hourly reference prices, never comparing nominal amounts of different assets.
- Every motif family (chain, cycle, fan-in, fan-out, fan-out-in, bipartite, stack, multi-stage, random) has legitimate decoy scenarios of the same shape (about 1,080 decoy transactions in train).
- Of the 225 hard transactions, 25 are ambiguous ordinary payments whose labels are independent of anything observable. They rank near the median, as expected.
- Assumptions: asset identity, conversions, leg counts and currencies carry no signal (the organisers say so; equal rates in every family confirmed it), so we do not use them. We use the whole week (offline task).

## Method

**Design rule.** Features are computed within one split, because wallets are disjoint between splits. No wallet id, hash, issuer, asset identity, ledger number, calendar date or split indicator is a feature. Labels and scenario ids never reach feature code. Thresholds and weights live in `config/params.yaml`. Every run is deterministic (`SEED = 42`, `n_jobs = 1`).

### Features (all hand-built from behaviour)

- `starter_`: a port of the organisers' notebook rules (pass-through within an hour, bursts from one sender).
- `tx_`: value, round-number flags, fee, hour of day, sender z-score and gaps. No asset or conversion fields.
- `acct_`: point-in-time sender and receiver aggregates over 1h, 24h and 72h windows.
- `flow_`: multi-hour timing. How soon and how much of the received money moves on, and how recently the sender was funded. This targets the finding that suspicious activity moves money faster than its look-alikes.
- `motif_`: temporal motif reconstruction. Payments are linked when money continues or a wallet sends or receives again within 10 minutes. We describe each linked component (size, span, rate, distinct wallets), plus chain depth, value retention and cycles.
- `shapes_`: signals taken from the organisers' training deck. These are the number of depositors, exit value against the sum of deposits, number of recipients, hub imbalance and overlap, and two-hop routes to the same destination.
- `graph_`, `rule_`: sender-receiver graph statistics; typology flags (structuring, pass-through, round amount, velocity burst).

### Model

LightGBM, supervised, with small regularised trees (7 leaves, at least 80 samples per leaf, 500 trees, learning rate 0.05, row and column subsampling, L2 5.0). The final score averages 5 seeds. Hyperparameter variants stayed within 0.4 AP of each other. An IsolationForest and a rule score exist in the code but carry weight 0 in the supervised ensemble.

### Post-processing

Many missed easy positives were the edges of bursts the model had already caught (fan-out leaves that keep the money, source payments). The `component_lift` step ranks the burst rather than the single payment. The final score is 0.5 x own rank + 0.5 x best rank in the transaction's 10-minute component. It is label-free. It is the v5 candidate (CV about 54) and is still being tested on the leaderboard.

### Rejected: learned scenario links

`src/model/links.py` trains a pair model on which nearby payments belong to the same scenario. It gave a +5 CV gain (to 54.9) but nothing on the leaderboard. It is disabled by default and not part of the submitted method.

## Validation & results

CV is 5 folds x 3 seeds, **grouped by scenario** so one scenario never sits in both fit and check parts. We compare changes **paired on identical folds** and keep one only if the gain exceeds 2 standard errors. Before keeping a feature we run a **KS drift check** between train and test and drop anything above 0.1. We allow **at most one leaderboard run per hypothesis**.

| Submission | Commit | CV AP | Leaderboard | Notes |
|---|---|---|---|---|
| Organisers' starter (logistic regression) | n/a | 30.0 | not submitted | baseline |
| v1 features | d13e6d0 | 50.1 | 57.3 | top-1% P 0.90, R 0.29 |
| v2 robust learned links | c877469 | 54.2 | 57.3 | identical to v1: links added nothing on test |
| v3 full learned links | c877469 (dirty) | 54.9 | 56.4 | CV gain did not transfer; links disabled |
| v4 shape features + 5-seed bagging | 2d0e667 | 52.2 | **57.5** | top-1% P 0.91; current best |
| v5 burst-level score lift | [commit TBD] | about 54 | [v5 result: TBD] | being tested now |

Fold AP for v1 ranged from 41 to 60, so leaderboard values sit inside the CV spread. By difficulty (v1 model): easy 68.5, medium 5.4, hard 0.9.

**What the CV-versus-leaderboard gap tells us.** Our CV groups by scenario, but all train scenarios come from one generator style, while the test set has batch variations. A model that learns train's scenario layout (the link model) gains in CV and gains nothing on test. Hand-built behavioural features transferred: v1 and the shape features in v4. The selection rule since then is hand-built features, a paired CV gain, no drift, then one leaderboard run to confirm. CV is a filter, not a forecast, and the leaderboard is one noisy test week.

## What the model flags and why

The model rewards speed and shape. Suspicious motifs complete in a median 0.13 hours against 1.0 hour for decoys (scenario-level AUC 0.79). Easy motifs take about 5 minutes against about an hour (AUC 0.91). The strongest gains came from `flow_` (+2.1 AP), `motif_` (+4.0) and `shapes_` (+1.7), all paired on identical folds.

Typical high scores: a sender funded by several depositors that pays out within minutes about the sum of those deposits (fan-in); one wallet paying many recipients in a short burst, or a chain forwarding a similar value hop by hop within minutes; several two-hop routes through middlemen to one destination (scatter-gather).

We did not compute per-feature importances; this attribution rests on the paired ablations.

## Uncertainty & false positives

Scores are rankings, not calibrated probabilities. The base rate is 3.1%, so a score is not the chance that a payment is criminal.

False positives crowd the top 1-3% of scores. In the top 3%, most false positives are legitimate relay-cohort payments (about 280) and busy background wallets (about 100). Legitimate decoy motifs of the same shape are a smaller share. A legitimate relay hub looks like a laundering hub, and a busy commercial wallet looks like a fan-out. Our ceiling analysis says that removing the worst quarter of relay false positives would add about 5 AP. Features aimed at them (pass-through share, hub share) did not change the top-3% composition.

Fold AP varies by about +/-5 within train alone, and the leaderboard moved by only 0 to 1.1 between variants, so such differences are noise.

## What is NOT caught

- **Medium (AP about 5-8).** These are the same motifs spread over 1-3 hours, so they are no faster than decoys and sit inside busy legitimate traffic. Only 27 such scenarios (225 transactions) exist in train. A specialist model on medium positives scored medium AP 1.2 (main model 6.3). Blending it in dragged overall AP to 30-40, and weighting medium positives 3x or 6x left medium unchanged. All three were rejected. Learning from 27 scenarios is also the train-specific signal that failed to transfer in v2 and v3.
- **Hard relays (AP about 1).** A hub collects from several wallets and pays out in waves over about 6 days, inside 12 legitimate cohorts that are exactly 1% suspicious. Only 12 suspicious relay scenarios exist in train. Faster hub forwarding (about 3 hours against about 30) is the only hint we found, and it is too weak to rank on.
- **Ambiguous (25 per full split).** Their labels are independent of observable features, so no method can learn them.

Medium and hard are 36% of the positives. Top-1% recall is only 0.29 because the top 1% (about 400 payments) is smaller than the number of positives.

## Data limitations & bias

- Synthetic data: planted patterns come from a generator and may not resemble real laundering, which adapts to detection.
- Leakage: wallets are reused across scenarios (487 of 700 appear in some suspicious motif), so wallet memory would look useful in CV and fail on test; no identity is a feature. A raw-field scan (sequence, ledger index, flags, SendMax, paths, fees) found no generator artifacts (AUC near 0.5).
- Features use the whole week, forward windows included: allowed in this offline task, unusable for real-time monitoring.
- Representativeness: one generator style and 700 wallets. Nothing here supports claims about real XRPL activity or real people. A flagged payment means "resembles a planted pattern", never "criminal".

## Reproducibility

`./run.sh` (organiser folders in `data/raw/{train,validation,test}`) writes and validates `outputs/predictions.csv`; `./run.sh --sample` uses bundled mini-splits. `make cv` runs grouped CV, `make test` the tests (including a byte-identical two-run check), `make zip` the upload archive without data. Python 3.13, LightGBM 4.7.0, versions pinned in `requirements.txt`. The full run takes about 2.5 minutes on a laptop CPU with 5-seed bagging, with no GPU and no network. Seeds come from `config/params.yaml` and `src/config.py`. Hardware is listed in the README. [clean-clone check on a second machine: TBD]

## What we'd do next

- Build features for the relay-hub false positives (balance dwell over days, cohort-level regularity) and test each one with the same paired, drift-checked rule.
- Test whether the burst-level lift in v5 transfers, then decide whether to keep it.
- Add per-transaction reasons (the CSV accepts a `reason` column) so an analyst can review flags.
