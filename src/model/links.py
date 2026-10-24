"""Learned scenario links (features prefixed `lk_`). Owner: P3/P4.

Fixed time rules can't rebuild motifs at hour scale (busy wallets merge everything), so we
LEARN which nearby payment pairs belong to the same scenario:

1. candidate pairs (earlier j -> later k, at most `window_secs` apart): continuation
   (j.receiver == k.sender), same sender, same receiver;
2. a LightGBM link model predicts "same scenario" from time gap, value ratio, recency rank
   and how many candidates there were (wallet busyness). Training targets come from
   `meta_scenario_id` (decoy and relay scenarios included); scenario ids are never features;
3. union-find on links with p >= threshold -> components -> size / span / wallets /
   pass-through / rate. Raw link probabilities are NOT features: they run lower on test
   (KS > 0.1), while components at p >= 0.6 stay stable.

Leakage control: train rows get OUT-OF-FOLD link probabilities (folds grouped by scenario),
so no tx is linked by a model that saw its own scenario; the target split uses a model fit
on all train pairs. Grouped CV 24 Oct: +5.1 ± 0.65 AP, 14/15 folds up.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.features.base import validate

log = logging.getLogger(__name__)
PREFIX = "lk_"
PAIR_FEATURES = ["kind", "log_dt", "rank", "n_cand", "log_ratio", "abs_ratio", "log_vj", "log_vk"]


def candidate_pairs(tx: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(time-sorted tx, pairs). Pair kinds: 0 continuation, 1 same sender, 2 same receiver.
    `rank` 0 = the most recent candidate of that kind; `n_cand` = candidates in the window."""
    tx = tx.sort_values(["ts", "tx_id"], kind="mergesort").reset_index(drop=True)
    t, v = tx["ts"].to_numpy(), tx["value_usd"].to_numpy()
    by_r, by_s = tx.groupby("receiver").indices, tx.groupby("sender").indices
    js, ks, kinds, ranks, ncand = [], [], [], [], []
    for kind, key, index in [(0, tx["sender"].to_numpy(), by_r), (1, tx["sender"].to_numpy(), by_s),
                             (2, tx["receiver"].to_numpy(), by_r)]:
        for k in range(len(tx)):
            idx = index.get(key[k])
            if idx is None:
                continue
            lo = np.searchsorted(t[idx], t[k] - window, "left")
            hi = np.searchsorted(t[idx], t[k], "left")
            cand = idx[lo:hi]
            cand = cand[cand != k][::-1]
            js.extend(cand)
            ks.extend([k] * len(cand))
            kinds.extend([kind] * len(cand))
            ranks.extend(range(len(cand)))
            ncand.extend([len(cand)] * len(cand))
    P = pd.DataFrame({"j": np.array(js, dtype=np.int64), "k": np.array(ks, dtype=np.int64),
                      "kind": kinds, "rank": ranks, "n_cand": ncand})
    P["log_dt"] = np.log1p(t[P["k"]] - t[P["j"]])
    P["log_ratio"] = np.log(v[P["k"]] / v[P["j"]])
    P["abs_ratio"] = P["log_ratio"].abs()
    P["log_vj"], P["log_vk"] = np.log1p(v[P["j"]]), np.log1p(v[P["k"]])
    return tx, P


def _model(cfg: dict):
    import lightgbm as lgb
    return lgb.LGBMClassifier(**cfg["features"]["links"]["model"], random_state=cfg["seed"],
                              n_jobs=cfg.get("n_jobs", 1))


def _targets(tx: pd.DataFrame, P: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(same-scenario 0/1 per pair, CV group per pair = scenario of k, or k itself)."""
    sc = tx["meta_scenario_id"].to_numpy()
    sj, sk = sc[P["j"]], sc[P["k"]]
    y = (pd.notna(sk) & (sj == sk)).astype(int)
    groups = pd.Series(sk).fillna(pd.Series(tx["tx_id"].to_numpy()[P["k"]])).to_numpy()
    return np.asarray(y), groups


def fit(tx_train: pd.DataFrame, cfg: dict):
    """Link model on all train pairs, plus train link features from OUT-OF-FOLD probabilities.
    Returns (model, train link features indexed by tx_id)."""
    lcfg = cfg["features"]["links"]
    txs, P = candidate_pairs(tx_train, int(lcfg["window_secs"]))
    y, groups = _targets(txs, P)
    oof = np.zeros(len(P))
    for fit_idx, chk_idx in GroupKFold(n_splits=cfg["cv"]["n_splits"]).split(P, y, groups):
        m = _model(cfg).fit(P.iloc[fit_idx][PAIR_FEATURES], y[fit_idx])
        oof[chk_idx] = m.predict_proba(P.iloc[chk_idx][PAIR_FEATURES])[:, 1]
    from sklearn.metrics import roc_auc_score
    log.info("links: %d pairs, %.2f%% same-scenario, OOF AUC %.3f", len(P), 100 * y.mean(),
             roc_auc_score(y, oof))
    model = _model(cfg).fit(P[PAIR_FEATURES], y)
    return model, features(txs, P, oof, lcfg["thresholds"], lcfg.get("prob_features", False))


def transform(model, tx: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Link features for an unlabelled split with the fitted link model."""
    lcfg = cfg["features"]["links"]
    txs, P = candidate_pairs(tx, int(lcfg["window_secs"]))
    p = model.predict_proba(P[PAIR_FEATURES])[:, 1] if len(P) else np.zeros(0)
    return features(txs, P, p, lcfg["thresholds"], lcfg.get("prob_features", False))


def features(txs: pd.DataFrame, P: pd.DataFrame, p: np.ndarray, thresholds,
             prob_features: bool = False) -> pd.DataFrame:
    """Component features per tx (txs is time-sorted, P indexes into it); with
    prob_features also each tx's strongest / summed link probabilities (drift on test)."""
    n = len(txs)
    t, snd, rcv = txs["ts"].to_numpy(), txs["sender"].to_numpy(), txs["receiver"].to_numpy()
    j, k = P["j"].to_numpy(), P["k"].to_numpy()
    f = pd.DataFrame(index=pd.Index(txs["tx_id"], name="tx_id"))
    for thr in thresholds:
        parent = np.arange(n)

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in zip(j[p >= thr], k[p >= thr]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        comp = np.array([find(i) for i in range(n)])
        d = pd.DataFrame({"c": comp, "t": t, "s": snd, "r": rcv})
        g = d.groupby("c")
        size = g["t"].transform("size").to_numpy()
        span = (g["t"].transform("max") - g["t"].transform("min")).to_numpy()
        shape = g.apply(lambda x: pd.Series({"wallets": len(set(x["s"]) | set(x["r"])),
                                             "through": len(set(x["s"]) & set(x["r"]))}),
                        include_groups=False)
        tag = f"{round(thr * 100)}"
        f[f"lk_size_{tag}"] = size
        f[f"lk_log_span_{tag}"] = np.log1p(span)
        f[f"lk_wallets_{tag}"] = shape["wallets"].reindex(comp).to_numpy()
        f[f"lk_through_{tag}"] = shape["through"].reindex(comp).to_numpy()
        f[f"lk_rate_{tag}"] = size / (span / 3600 + 0.1)
    if prob_features:
        pin = pd.Series(p).groupby(k).agg(["max", "sum"]).reindex(range(n)).fillna(0)
        pout = pd.Series(p).groupby(j).agg(["max", "sum"]).reindex(range(n)).fillna(0)
        f["lk_p_in_max"], f["lk_p_in_sum"] = pin["max"].to_numpy(), pin["sum"].to_numpy()
        f["lk_p_out_max"], f["lk_p_out_sum"] = pout["max"].to_numpy(), pout["sum"].to_numpy()
    return validate(f, PREFIX)
