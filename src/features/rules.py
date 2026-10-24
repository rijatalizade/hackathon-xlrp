"""Typology rule flags (prefix `rule_`). Owner: P5. Thresholds live in params.yaml `rules`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import rolling_count, to_seconds, validate

PREFIX = "rule_"


def structuring(tx: pd.DataFrame, secs: np.ndarray, p: dict) -> np.ndarray:
    """1 if the tx is just below threshold T and its sender made >= N such txs within
    +/- W seconds of it."""
    T, margin = p["threshold"], p["margin"]
    cand = ((tx["amount"] >= T * (1 - margin)) & (tx["amount"] < T)).to_numpy()
    out = np.zeros(len(tx), dtype=bool)
    if cand.any():
        s, t, w = tx["sender"].to_numpy()[cand], secs[cand], int(p["window_secs"])
        both = rolling_count(s, t, w) + rolling_count(s, -t, w) - 1
        out[cand] = both >= p["min_count"]
    return out


def passthrough(tx: pd.DataFrame, secs: np.ndarray, p: dict) -> np.ndarray:
    """1 if the receiver sends out >= forward_ratio * amount within (t, t + W]."""
    w = int(p["window_secs"])
    accts = pd.Index(np.unique(np.concatenate([tx["sender"], tx["receiver"]])))
    s_code = accts.get_indexer(tx["sender"]).astype(np.int64)
    r_code = accts.get_indexer(tx["receiver"]).astype(np.int64)
    rel = secs - secs.min()
    span = int(rel.max()) + w + 1
    order = np.lexsort((rel, s_code))
    out_key = s_code[order] * span + rel[order]
    cum = np.concatenate([[0.0], np.cumsum(tx["amount"].to_numpy()[order])])
    lo = r_code * span + rel
    fwd = (cum[np.searchsorted(out_key, lo + w, side="right")]
           - cum[np.searchsorted(out_key, lo, side="right")])
    amt = tx["amount"].to_numpy()
    return (amt > 0) & (fwd >= p["forward_ratio"] * amt)


def round_amount(tx: pd.DataFrame, p: dict) -> np.ndarray:
    """1 if amount is a non-zero multiple of `level`."""
    return ((tx["amount"] > 0) & (tx["amount"] % p["level"] == 0)).to_numpy()


def velocity_burst(tx: pd.DataFrame, secs: np.ndarray, p: dict) -> np.ndarray:
    """1 if the sender has >= min_count txs in the short window and that rate is
    >= rate_ratio x its long-window rate."""
    s = tx["sender"].to_numpy()
    n_short = rolling_count(s, secs, int(p["short_secs"]))
    n_long = rolling_count(s, secs, int(p["long_secs"]))
    expected = n_long * p["short_secs"] / p["long_secs"]
    return (n_short >= p["min_count"]) & (n_short >= p["rate_ratio"] * expected)


def build(tx: pd.DataFrame, accounts: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """All rule flags plus `rule_score` = weighted mean of flags (weights in params)."""
    rc = cfg["rules"]
    secs = to_seconds(tx["timestamp"])
    f = pd.DataFrame(index=pd.Index(tx["tx_id"], name="tx_id"))
    f["rule_structuring"] = structuring(tx, secs, rc["structuring"])
    f["rule_passthrough"] = passthrough(tx, secs, rc["passthrough"])
    f["rule_round_amount"] = round_amount(tx, rc["round_amount"])
    f["rule_velocity_burst"] = velocity_burst(tx, secs, rc["velocity_burst"])
    # TODO(P5): add rule_currency_hop, rule_fan_in_fan_out, rule_new_account once EDA confirms.
    weights = {k: v for k, v in rc.get("weights", {}).items() if k in f}
    total = sum(weights.values()) or 1.0
    f["rule_score"] = sum(f[k].astype(float) * w for k, w in weights.items()) / total
    return validate(f, PREFIX)
