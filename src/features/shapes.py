"""Hand-built laundering-shape signals from the organisers' training deck (prefix `shape_`).
Owner: P2/P3.

Per shape slide, "signals to compute" we did not have yet, all within one split and
time-local (windows in params.yaml `features.shapes`):

- fan-in:          distinct depositors into the sender just before it pays; exit value
                   vs the sum of those deposits
- fan-out:         distinct recipients the sender pays around now (payout similarity was
                   dropped: it drifts between train and test, KS 0.15)
- gather-scatter:  the receiver keeps little balance (|in - out| / (in + out) around now);
                   its depositors and payees never overlap
- scatter-gather:  several two-hop routes sender -> middleman -> same destination
Learned nothing from labels: pure behaviour, so expected to transfer to test like v1 did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import validate

PREFIX = "shape_"


def _index(tx: pd.DataFrame, col: str):
    """{wallet: (times, values, counterparties)} for txs where wallet is `col`, time-sorted."""
    t, v = tx["ts"].to_numpy(), tx["value_usd"].to_numpy()
    other = tx["receiver" if col == "sender" else "sender"].to_numpy()
    return {a: (t[i], v[i], other[i]) for a, i in tx.groupby(col, sort=False).indices.items()}


def _window(entry, lo, hi):
    """Slice of (times, values, counterparties) with lo <= time <= hi."""
    if entry is None:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=object)
    t, v, o = entry
    a, b = np.searchsorted(t, lo, "left"), np.searchsorted(t, hi, "right")
    return t[a:b], v[a:b], o[a:b]


def build(tx: pd.DataFrame, legs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    scfg = cfg["features"]["shapes"]
    tx = tx.sort_values(["ts", "tx_id"], kind="mergesort").reset_index(drop=True)
    t, v = tx["ts"].to_numpy(), tx["value_usd"].to_numpy()
    S, R = tx["sender"].to_numpy(), tx["receiver"].to_numpy()
    out_of, in_of = _index(tx, "sender"), _index(tx, "receiver")
    n = len(tx)
    f = pd.DataFrame(index=pd.Index(tx["tx_id"], name="tx_id"))

    for name, w in scfg["windows_secs"].items():
        w = int(w)
        s_dep, s_exit_ratio = np.zeros(n), np.full(n, np.nan)
        s_rec = np.zeros(n)
        r_imb, r_overlap, r_n_in, r_n_out = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), np.zeros(n)
        for k in range(n):
            ti, vi, s, r = t[k], v[k], S[k], R[k]
            # fan-in: who fed the sender before this payment, and does it pay out the total?
            _, dv, ds = _window(in_of.get(s), ti - w, ti - 1)
            if len(dv):
                s_dep[k] = len(set(ds))
                s_exit_ratio[k] = vi / dv.sum()
            # fan-out: the sender's payouts around now
            _, ov, orc = _window(out_of.get(s), ti - w, ti + w)
            if len(ov):
                s_rec[k] = len(set(orc))
            # gather-scatter: the receiver as a hub around now
            _, iv, isn = _window(in_of.get(r), ti - w, ti + w)
            _, rv, rrc = _window(out_of.get(r), ti - w, ti + w)
            r_n_in[k], r_n_out[k] = len(set(isn)), len(set(rrc))
            tot = iv.sum() + rv.sum()
            if tot > 0:
                r_imb[k] = abs(iv.sum() - rv.sum()) / tot
            if len(isn) and len(rrc):
                a, b = set(isn), set(rrc)
                r_overlap[k] = len(a & b) / len(a | b)
        f[f"shape_s_depositors_{name}"] = s_dep
        f[f"shape_s_log_exit_over_in_{name}"] = np.log(s_exit_ratio)
        f[f"shape_s_recipients_{name}"] = s_rec
        f[f"shape_r_imbalance_{name}"] = r_imb
        f[f"shape_r_overlap_{name}"] = r_overlap
        f[f"shape_r_depositors_{name}"] = r_n_in
        f[f"shape_r_payees_{name}"] = r_n_out

    # scatter-gather: routes s -> m -> d; count distinct middlemen m per (s, d) within w2
    w2 = int(scfg["two_hop_window_secs"])
    routes = {}
    for k in range(n):
        _, _, dest = _window(out_of.get(R[k]), t[k] + 1, t[k] + w2)
        for d in set(dest):
            routes.setdefault((S[k], d), set()).add(R[k])
    two_hop = np.zeros(n)
    for k in range(n):
        _, _, dest = _window(out_of.get(R[k]), t[k] + 1, t[k] + w2)
        if len(dest):
            two_hop[k] = max(len(routes[(S[k], d)]) for d in set(dest))
    f["shape_two_hop_routes"] = two_hop
    return validate(f, PREFIX)
